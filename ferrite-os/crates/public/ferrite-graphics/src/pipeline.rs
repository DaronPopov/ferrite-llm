//! Render pipeline — wires Camera, Light, and Mesh into RenderContext.
//!
//! This module bridges the high-level ferrite-graphics types with the
//! low-level ptx-render kernel stack. No temporary Tensor allocations —
//! everything goes through the RenderContext's scratch arena.

use ptx_render::RenderContext;

use crate::camera::Camera;
use crate::light::Light;
use crate::mesh::Mesh;
use crate::Result;

/// Run the full render pipeline for one mesh draw call.
///
/// Uses the RenderContext's pre-allocated GPU memory:
/// - MVP matrix → persistent pipeline state (uploaded, not allocated)
/// - Clip positions → scratch arena (bump, zero cost)
/// - Screen positions → scratch arena (bump, zero cost)
/// - Light parameters → persistent pipeline state
pub fn draw(
    ctx: &mut RenderContext,
    mesh: &Mesh,
    camera: &Camera,
    light: Option<&Light>,
) -> Result<()> {
    let num_verts = mesh.num_vertices() as u32;
    let num_tris = mesh.num_triangles() as u32;

    // ── Upload MVP matrix (overwrites persistent GPU buffer, no alloc) ──
    ctx.set_camera(camera.view_projection_matrix())?;

    // ── Upload light parameters if provided ─────────────────────────────
    if let Some(light) = light {
        let (dir, color, ambient) = light.params();
        ctx.set_light(dir, color, ambient)?;
    }

    // ── Draw: vertex transform + viewport + rasterize + normals ─────────
    let positions = mesh.positions.data_ptr_typed::<f32>() as *const f32;
    let normals = mesh
        .normals
        .as_ref()
        .map(|n| n.data_ptr_typed::<f32>() as *const f32)
        .unwrap_or(std::ptr::null());
    let colors = mesh
        .colors
        .as_ref()
        .map(|c| c.data_ptr_typed::<f32>() as *const f32)
        .unwrap_or(std::ptr::null());
    let indices = mesh.indices.data_ptr_typed::<u32>() as *const u32;

    ctx.draw_indexed(
        positions,
        normals,
        colors,
        indices,
        num_verts,
        num_tris,
    )?;

    Ok(())
}
