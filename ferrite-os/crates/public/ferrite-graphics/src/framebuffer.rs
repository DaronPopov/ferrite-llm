//! Framebuffer — GPU-resident color + depth buffers backed by ptx-render.
//!
//! Wraps [`ptx_render::RenderContext`] for zero per-frame TLSF allocations.
//! The scratch arena handles all temporary vertex data; persistent framebuffer
//! memory is allocated once at creation.

use std::sync::Arc;

use ptx_render::{RenderConfig, RenderContext};
use ptx_runtime::PtxRuntime;

use crate::Result;

/// A GPU-resident framebuffer with color, depth, and optional normal buffers.
///
/// Internally powered by [`ptx_render::RenderContext`] — all GPU memory is
/// pre-allocated from the TLSF pool at creation. Per-frame rendering uses
/// only bump allocation (zero TLSF overhead).
pub struct Framebuffer {
    /// The internal render context (owns all GPU memory).
    ctx: RenderContext,
    /// Framebuffer width in pixels.
    width: u32,
    /// Framebuffer height in pixels.
    height: u32,
}

impl Framebuffer {
    /// Create a new framebuffer.
    ///
    /// Pre-allocates all GPU memory (color, depth, normal, RGBA staging,
    /// scratch arena) from the TLSF pool. After this call, rendering is
    /// zero-alloc on the GPU memory pool.
    pub fn new(width: u32, height: u32, runtime: &Arc<PtxRuntime>) -> Result<Self> {
        let config = RenderConfig {
            max_vertices: 100_000,
            fb_width: width,
            fb_height: height,
        };
        let ctx = RenderContext::new(config, runtime)?;

        Ok(Self { ctx, width, height })
    }

    /// Number of pixels.
    pub fn num_pixels(&self) -> u32 {
        self.ctx.num_pixels()
    }

    /// Framebuffer dimensions.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Clear the framebuffer to a solid color and reset depth.
    ///
    /// Also resets the scratch arena (zero cost) for the new frame.
    pub fn clear(&mut self, color: [f32; 4]) {
        self.ctx.begin_frame(color);
    }

    /// Draw a mesh with camera and optional lighting.
    ///
    /// Runs the full pipeline: vertex transform → viewport → rasterize →
    /// optional normal rasterization. All temporary GPU memory comes from
    /// the pre-allocated scratch arena (zero TLSF alloc/free per call).
    pub fn draw_mesh(
        &mut self,
        mesh: &crate::Mesh,
        camera: &crate::Camera,
        light: Option<&crate::Light>,
        _runtime: &Arc<PtxRuntime>,
    ) -> Result<()> {
        crate::pipeline::draw(&mut self.ctx, mesh, camera, light)
    }

    /// Convert the color buffer to packed `0xRRGGBBAA` u32 for `ferrite-window::present()`.
    ///
    /// Runs shading (if lighting was set), converts float RGBA → packed u32,
    /// synchronizes the GPU, and copies pixels back to host memory.
    ///
    /// Returns a slice valid until the next `clear()` / `to_rgba()` call.
    pub fn to_rgba(&mut self, _runtime: &Arc<PtxRuntime>) -> Result<Vec<u32>> {
        let pixels = self.ctx.end_frame()?;
        Ok(pixels.to_vec())
    }

    /// Scratch arena usage for debugging.
    pub fn scratch_stats(&self) -> (usize, usize) {
        self.ctx.scratch_stats()
    }
}

impl std::fmt::Debug for Framebuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Framebuffer({}x{}, {:?})",
            self.width, self.height, self.ctx
        )
    }
}
