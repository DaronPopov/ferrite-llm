//! Typed kernel dispatch — safe wrappers around ptx_sys graphics FFI.
//!
//! **Level 1** of the render stack. Each function is a thin wrapper that
//! takes typed GPU pointers and a CUDA stream, dispatches a single kernel,
//! and returns immediately (async on GPU).
//!
//! No allocation, no synchronization, no state — pure dispatch.

use ptx_sys::cudaStream_t;

/// Clear a float RGBA color buffer to a solid color.
#[inline]
pub unsafe fn clear_color(
    buf: *mut f32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
    num_pixels: u32,
    stream: cudaStream_t,
) {
    ptx_sys::ptx_gfx_clear_color(buf, r, g, b, a, num_pixels, stream);
}

/// Clear an integer depth buffer (write +inf as u32 for far plane).
#[inline]
pub unsafe fn clear_depth(buf: *mut f32, num_pixels: u32, stream: cudaStream_t) {
    ptx_sys::ptx_gfx_clear_depth(buf, f32::INFINITY, num_pixels, stream);
}

/// Transform vertex positions by a 4x4 MVP matrix.
///
/// * `pos_in`  — `[N * 3]` f32 (object-space xyz)
/// * `pos_out` — `[N * 4]` f32 (clip-space xyzw)
/// * `mvp`     — `[16]` f32 (column-major 4x4)
#[inline]
pub unsafe fn vertex_transform(
    pos_in: *const f32,
    pos_out: *mut f32,
    mvp: *const f32,
    num_vertices: u32,
    stream: cudaStream_t,
) {
    ptx_sys::ptx_gfx_vertex_transform(pos_in, pos_out, mvp, num_vertices, stream);
}

/// Perspective divide + viewport map: clip-space → screen-space.
///
/// * `clip` — `[N * 4]` f32 (clip xyzw)
/// * `screen` — `[N * 3]` f32 (screen_x, screen_y, depth)
#[inline]
pub unsafe fn viewport_transform(
    clip: *const f32,
    screen: *mut f32,
    vp_w: u32,
    vp_h: u32,
    num_vertices: u32,
    stream: cudaStream_t,
) {
    ptx_sys::ptx_gfx_viewport_transform(clip, screen, vp_w, vp_h, num_vertices, stream);
}

/// Rasterize indexed triangles with atomic depth test.
///
/// * `screen_pos` — `[V * 3]` f32 (screen-space)
/// * `colors`     — `[V * 3]` f32 or null (vertex colors)
/// * `indices`    — `[T * 3]` u32
/// * `color_buf`  — `[H * W * 4]` f32 (RGBA output)
/// * `depth_buf`  — `[H * W]` u32 (integer depth for atomicMin)
#[inline]
pub unsafe fn rasterize_triangles(
    screen_pos: *const f32,
    colors: *const f32,
    indices: *const u32,
    color_buf: *mut f32,
    depth_buf: *mut u32,
    fb_w: u32,
    fb_h: u32,
    num_triangles: u32,
    stream: cudaStream_t,
) {
    ptx_sys::ptx_gfx_rasterize_triangles(
        screen_pos, colors, indices, color_buf, depth_buf, fb_w, fb_h, num_triangles, stream,
    );
}

/// Rasterize interpolated normals (uses existing depth buffer for matching).
#[inline]
pub unsafe fn rasterize_normals(
    screen_pos: *const f32,
    vtx_normals: *const f32,
    indices: *const u32,
    normal_buf: *mut f32,
    depth_buf: *const u32,
    fb_w: u32,
    fb_h: u32,
    num_triangles: u32,
    stream: cudaStream_t,
) {
    ptx_sys::ptx_gfx_rasterize_normals(
        screen_pos, vtx_normals, indices, normal_buf, depth_buf, fb_w, fb_h, num_triangles, stream,
    );
}

/// Per-pixel Blinn-Phong shading (modifies color buffer in-place).
#[inline]
pub unsafe fn shade_phong(
    normal_buf: *const f32,
    color_buf: *mut f32,
    depth_buf: *const u32,
    light_dir: *const f32,
    light_color: *const f32,
    ambient: *const f32,
    num_pixels: u32,
    stream: cudaStream_t,
) {
    ptx_sys::ptx_gfx_shade_phong(
        normal_buf, color_buf, depth_buf, light_dir, light_color, ambient, num_pixels, stream,
    );
}

/// Convert float RGBA framebuffer to packed 0xRRGGBBAA u32.
#[inline]
pub unsafe fn framebuffer_to_rgba(
    color_buf: *const f32,
    rgba_out: *mut u32,
    num_pixels: u32,
    stream: cudaStream_t,
) {
    ptx_sys::ptx_gfx_framebuffer_to_rgba(color_buf, rgba_out, num_pixels, stream);
}
