/// OpenCV-backed camera capture with TLSF frame buffers (requires `capture` feature).

pub mod camera;
pub mod convert;
pub mod frame;

#[cfg(feature = "torch")]
pub mod bridge;
