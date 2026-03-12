use std::ptr::NonNull;
use std::sync::Arc;

use crate::cpu_tlsf::CpuTlsf;
use crate::runtime::context::HasAllocator;
use crate::{LangError, Result};

/// Pixel format of the frame buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    Bgr8,
    Rgb8,
    Gray8,
}

impl PixelFormat {
    /// Bytes per pixel for this format.
    pub fn channels(self) -> usize {
        match self {
            PixelFormat::Bgr8 | PixelFormat::Rgb8 => 3,
            PixelFormat::Gray8 => 1,
        }
    }
}

/// Metadata describing a single captured frame.
#[derive(Clone, Debug)]
pub struct FrameMeta {
    pub width: usize,
    pub height: usize,
    pub format: PixelFormat,
    pub frame_index: u64,
    pub timestamp_us: u64,
}

impl FrameMeta {
    /// Total number of bytes needed for pixel data.
    pub fn byte_len(&self) -> usize {
        self.width * self.height * self.format.channels()
    }
}

const FRAME_ALIGN: usize = 64; // SIMD-friendly alignment

/// A single video frame backed by the TLSF allocator.
///
/// Mirrors the `TlsfStorage<T>` pattern from `runtime/tensor.rs` but
/// specialised for `u8` pixel data with 64-byte alignment.
pub struct Frame {
    meta: FrameMeta,
    ptr: NonNull<u8>,
    len: usize,
    alloc: Arc<CpuTlsf>,
}

// SAFETY: The TLSF-backed pointer is exclusively owned by Frame;
// CpuTlsf is Mutex-protected and Send+Sync.
unsafe impl Send for Frame {}
unsafe impl Sync for Frame {}

impl Drop for Frame {
    fn drop(&mut self) {
        // SAFETY: ptr/len/align originate from the same allocator.
        let _ = unsafe { self.alloc.deallocate(self.ptr, self.len, FRAME_ALIGN) };
    }
}

impl Frame {
    /// Allocate an uninitialised frame from TLSF.
    pub fn allocate(ctx: &(impl HasAllocator + ?Sized), meta: FrameMeta) -> Result<Self> {
        let len = meta.byte_len();
        let ptr = ctx.allocator().allocate(len.max(1), FRAME_ALIGN)?;
        Ok(Self {
            meta,
            ptr,
            len,
            alloc: Arc::clone(ctx.allocator()),
        })
    }

    /// Allocate and copy `data` into a new TLSF frame.
    pub fn from_bytes(ctx: &(impl HasAllocator + ?Sized), meta: FrameMeta, data: &[u8]) -> Result<Self> {
        let len = meta.byte_len();
        if data.len() < len {
            return Err(LangError::Capture {
                message: format!(
                    "frame data too short: need {} bytes, got {}",
                    len,
                    data.len()
                ),
            });
        }
        let ptr = ctx.allocator().allocate(len.max(1), FRAME_ALIGN)?;
        // SAFETY: ptr is valid for len writable bytes; data.len() >= len.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.as_ptr(), len);
        }
        Ok(Self {
            meta,
            ptr,
            len,
            alloc: Arc::clone(ctx.allocator()),
        })
    }

    pub fn meta(&self) -> &FrameMeta {
        &self.meta
    }

    pub fn width(&self) -> usize {
        self.meta.width
    }

    pub fn height(&self) -> usize {
        self.meta.height
    }

    pub fn format(&self) -> PixelFormat {
        self.meta.format
    }

    /// Zero-copy view of the pixel data.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for len initialised bytes.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Mutable zero-copy view of the pixel data.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: exclusive access via &mut self; ptr valid for len bytes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Total byte length of pixel data.
    pub fn byte_len(&self) -> usize {
        self.len
    }
}

/// Pre-allocated pool of frames for zero-alloc capture loops.
///
/// `acquire()` borrows a frame from the pool; dropping it does **not**
/// return it automatically — call `release()` to recycle.
pub struct FramePool {
    frames: Vec<Frame>,
}

impl FramePool {
    /// Pre-allocate `count` frames of the given dimensions.
    pub fn new(
        ctx: &(impl HasAllocator + ?Sized),
        width: usize,
        height: usize,
        format: PixelFormat,
        count: usize,
    ) -> Result<Self> {
        let mut frames = Vec::with_capacity(count);
        for i in 0..count {
            let meta = FrameMeta {
                width,
                height,
                format,
                frame_index: i as u64,
                timestamp_us: 0,
            };
            frames.push(Frame::allocate(ctx, meta)?);
        }
        Ok(Self { frames })
    }

    /// Take a frame from the pool.  Returns `None` when exhausted.
    pub fn acquire(&mut self) -> Option<Frame> {
        self.frames.pop()
    }

    /// Return a frame to the pool for reuse.
    pub fn release(&mut self, frame: Frame) {
        self.frames.push(frame);
    }

    /// Number of frames currently available in the pool.
    pub fn available(&self) -> usize {
        self.frames.len()
    }
}
