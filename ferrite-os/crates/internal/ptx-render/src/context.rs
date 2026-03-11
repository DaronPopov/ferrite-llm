//! RenderContext — frame-level orchestrator for the GPU render stack.
//!
//! **Level 3** of the render stack. Owns all GPU memory (scratch arena +
//! persistent framebuffer) and exposes a begin/draw/end frame API.
//!
//! # Memory layout
//!
//! ```text
//! TLSF pool
//!   ├─ ScratchArena (one alloc, bump-allocated per frame)
//!   │    ├─ clip_positions [N*4] f32    ← per draw call
//!   │    ├─ screen_positions [N*3] f32  ← per draw call
//!   │    └─ ... (reset each frame)
//!   ├─ color_buf [W*H*4] f32           ← persistent
//!   ├─ depth_buf [W*H] u32             ← persistent
//!   ├─ normal_buf [W*H*3] f32          ← persistent
//!   ├─ rgba_buf [W*H] u32              ← persistent (readback staging)
//!   └─ PipelineState (mvp + light uniforms) ← persistent
//! ```
//!
//! Per-frame TLSF allocations: **zero**. The scratch arena bump-resets.

use std::sync::Arc;

use ptx_runtime::{GpuPtr, PtxRuntime, Result, Error};

use crate::arena::{GpuSlice, ScratchArena};
use crate::dispatch;
use crate::state::PipelineState;

/// Configuration for creating a render context.
#[derive(Debug, Clone)]
pub struct RenderConfig {
    /// Maximum vertices per draw call.
    pub max_vertices: u32,
    /// Framebuffer width.
    pub fb_width: u32,
    /// Framebuffer height.
    pub fb_height: u32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            max_vertices: 100_000,
            fb_width: 1920,
            fb_height: 1080,
        }
    }
}

/// GPU render context — owns all memory and orchestrates the render pipeline.
///
/// Created once. Pre-allocates everything from TLSF. Zero per-frame allocations.
pub struct RenderContext {
    // ── Scratch arena (per-frame temporaries) ──────────────────────
    scratch: ScratchArena,
    max_vertices: u32,

    // ── Persistent framebuffer memory ──────────────────────────────
    color_buf: GpuPtr,   // [H * W * 4] f32
    depth_buf: GpuPtr,   // [H * W] u32
    normal_buf: GpuPtr,  // [H * W * 3] f32
    rgba_buf: GpuPtr,    // [H * W] u32

    fb_width: u32,
    fb_height: u32,
    num_pixels: u32,

    // ── Pipeline state (GPU-side uniforms) ─────────────────────────
    state: PipelineState,

    // ── Host readback buffer ───────────────────────────────────────
    host_pixels: Vec<u32>,

    // ── Runtime + dedicated render stream ──────────────────────────
    runtime: Arc<PtxRuntime>,
}

impl RenderContext {
    /// Create a new render context. All GPU memory is allocated here.
    ///
    /// After this call, every subsequent operation is zero-alloc on the
    /// TLSF pool. The scratch arena uses bump allocation; persistent
    /// buffers are reused across frames.
    pub fn new(config: RenderConfig, runtime: &Arc<PtxRuntime>) -> Result<Self> {
        let np = config.fb_width * config.fb_height;

        // ── Scratch arena sizing ─────────────────────────────────────
        // Per draw call: clip_positions [N*4] + screen_positions [N*3] = N*28 bytes
        // Plus alignment padding. Budget for 4 draw calls per frame.
        let scratch_bytes =
            (config.max_vertices as usize) * 28 * 4 + 4096; // headroom
        let scratch = ScratchArena::new(scratch_bytes, runtime)?;

        // ── Persistent framebuffer allocations ───────────────────────
        let color_buf = runtime.alloc(np as usize * 4 * 4)?;  // 4 channels * f32
        let depth_buf = runtime.alloc(np as usize * 4)?;        // u32 per pixel
        let normal_buf = runtime.alloc(np as usize * 3 * 4)?;  // 3 channels * f32
        let rgba_buf = runtime.alloc(np as usize * 4)?;          // u32 per pixel

        // ── Pipeline state ───────────────────────────────────────────
        let state = PipelineState::new(config.fb_width, config.fb_height, runtime)?;

        // ── Host readback buffer ─────────────────────────────────────
        let host_pixels = vec![0u32; np as usize];

        Ok(Self {
            scratch,
            max_vertices: config.max_vertices,
            color_buf,
            depth_buf,
            normal_buf,
            rgba_buf,
            fb_width: config.fb_width,
            fb_height: config.fb_height,
            num_pixels: np,
            state,
            host_pixels,
            runtime: Arc::clone(runtime),
        })
    }

    /// Begin a new frame. Resets the scratch arena (zero cost) and clears
    /// the framebuffer on the GPU.
    pub fn begin_frame(&mut self, clear_color: [f32; 4]) {
        self.scratch.reset();
        self.state.lighting_enabled = false;

        let stream = self.stream();
        unsafe {
            dispatch::clear_color(
                self.color_ptr(),
                clear_color[0],
                clear_color[1],
                clear_color[2],
                clear_color[3],
                self.num_pixels,
                stream,
            );
            dispatch::clear_depth(
                self.depth_buf.as_ptr() as *mut f32,
                self.num_pixels,
                stream,
            );
        }
    }

    /// Set the camera (model-view-projection matrix) for subsequent draw calls.
    pub fn set_camera(&self, mvp: &[f32; 16]) -> Result<()> {
        self.state.set_mvp(mvp)
    }

    /// Set the directional light for subsequent draw calls.
    pub fn set_light(
        &mut self,
        dir: [f32; 3],
        color: [f32; 3],
        ambient: [f32; 3],
    ) -> Result<()> {
        self.state.set_light(dir, color, ambient)
    }

    /// Draw an indexed mesh.
    ///
    /// All pointers must be valid GPU device pointers (e.g., from
    /// `Tensor::data_ptr_typed()` or another TLSF allocation).
    ///
    /// - `positions` — `[N * 3]` f32, object-space vertex positions
    /// - `normals`   — `[N * 3]` f32, or null (skips normal rasterization)
    /// - `colors`    — `[N * 3]` f32, or null (white default)
    /// - `indices`   — `[T * 3]` u32, triangle index buffer
    pub fn draw_indexed(
        &mut self,
        positions: *const f32,
        normals: *const f32,
        colors: *const f32,
        indices: *const u32,
        num_vertices: u32,
        num_triangles: u32,
    ) -> Result<()> {
        if num_vertices > self.max_vertices {
            return Err(Error::Internal {
                message: format!(
                    "draw exceeds max_vertices: {} > {}",
                    num_vertices, self.max_vertices
                ),
            });
        }

        let stream = self.stream();

        // ── Scratch-allocate temporaries (bump, zero cost) ──────────
        let clip: GpuSlice<f32> = self
            .scratch
            .alloc(num_vertices as usize * 4)
            .ok_or_else(|| Error::Internal {
                message: "scratch arena exhausted for clip_positions".into(),
            })?;

        let screen: GpuSlice<f32> = self
            .scratch
            .alloc(num_vertices as usize * 3)
            .ok_or_else(|| Error::Internal {
                message: "scratch arena exhausted for screen_positions".into(),
            })?;

        // ── Geometry pass: vertex → clip → screen ───────────────────
        unsafe {
            dispatch::vertex_transform(
                positions,
                clip.as_ptr(),
                self.state.mvp_ptr(),
                num_vertices,
                stream,
            );

            dispatch::viewport_transform(
                clip.as_ptr(),
                screen.as_ptr(),
                self.fb_width,
                self.fb_height,
                num_vertices,
                stream,
            );
        }

        // ── Rasterization pass ──────────────────────────────────────
        unsafe {
            dispatch::rasterize_triangles(
                screen.as_ptr(),
                colors,
                indices,
                self.color_ptr(),
                self.depth_ptr(),
                self.fb_width,
                self.fb_height,
                num_triangles,
                stream,
            );
        }

        // ── Normal rasterization (if mesh has normals) ──────────────
        if !normals.is_null() {
            unsafe {
                dispatch::rasterize_normals(
                    screen.as_ptr(),
                    normals,
                    indices,
                    self.normal_ptr(),
                    self.depth_buf.as_ptr() as *const u32,
                    self.fb_width,
                    self.fb_height,
                    num_triangles,
                    stream,
                );
            }
        }

        Ok(())
    }

    /// Finalize the frame: run shading, convert to RGBA, read back to host.
    ///
    /// Returns the host pixel buffer (`0xRRGGBBAA` packed u32, row-major).
    /// This pointer is valid until the next `end_frame()` call.
    pub fn end_frame(&mut self) -> Result<&[u32]> {
        let stream = self.stream();

        // ── Shading pass ────────────────────────────────────────────
        if self.state.lighting_enabled {
            unsafe {
                dispatch::shade_phong(
                    self.normal_buf.as_ptr() as *const f32,
                    self.color_ptr(),
                    self.depth_buf.as_ptr() as *const u32,
                    self.state.light_dir_ptr(),
                    self.state.light_color_ptr(),
                    self.state.ambient_ptr(),
                    self.num_pixels,
                    stream,
                );
            }
        }

        // ── Framebuffer → packed RGBA ───────────────────────────────
        unsafe {
            dispatch::framebuffer_to_rgba(
                self.color_buf.as_ptr() as *const f32,
                self.rgba_buf.as_ptr() as *mut u32,
                self.num_pixels,
                stream,
            );
        }

        // ── Synchronize and read back to host ───────────────────────
        self.sync()?;

        let size_bytes = self.num_pixels as usize * 4;
        unsafe {
            self.rgba_buf.copy_to_host(
                self.host_pixels.as_mut_ptr() as *mut libc::c_void,
                size_bytes,
            )?;
        }

        Ok(&self.host_pixels)
    }

    /// Framebuffer dimensions.
    #[inline]
    pub fn size(&self) -> (u32, u32) {
        (self.fb_width, self.fb_height)
    }

    /// Number of pixels.
    #[inline]
    pub fn num_pixels(&self) -> u32 {
        self.num_pixels
    }

    /// Scratch arena stats for debugging.
    pub fn scratch_stats(&self) -> (usize, usize) {
        (self.scratch.used(), self.scratch.capacity())
    }

    /// Direct access to the color buffer (GPU pointer).
    #[inline]
    pub fn color_ptr(&self) -> *mut f32 {
        self.color_buf.as_ptr() as *mut f32
    }

    /// Direct access to the depth buffer (GPU pointer).
    #[inline]
    pub fn depth_ptr(&self) -> *mut u32 {
        self.depth_buf.as_ptr() as *mut u32
    }

    /// Direct access to the normal buffer (GPU pointer).
    #[inline]
    pub fn normal_ptr(&self) -> *mut f32 {
        self.normal_buf.as_ptr() as *mut f32
    }

    // ── Internals ────────────────────────────────────────────────────

    fn stream(&self) -> ptx_sys::cudaStream_t {
        self.runtime.next_stream().raw()
    }

    fn sync(&self) -> Result<()> {
        self.runtime.next_stream().synchronize()
    }
}

impl std::fmt::Debug for RenderContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RenderContext({}x{}, max_verts={}, scratch={}/{})",
            self.fb_width,
            self.fb_height,
            self.max_vertices,
            self.scratch.used(),
            self.scratch.capacity(),
        )
    }
}
