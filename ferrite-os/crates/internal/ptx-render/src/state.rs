//! Pipeline state — camera, lights, and viewport stored on GPU.
//!
//! **Level 2** of the render stack. Manages small uniform-like GPU buffers
//! for the MVP matrix, light parameters, and viewport dimensions.
//! These are uploaded once per `set_*` call, not per draw call.

use std::sync::Arc;

use ptx_runtime::{GpuPtr, PtxRuntime, Result};

/// GPU-resident pipeline state: camera matrix + light parameters.
///
/// Each buffer is a small, persistent TLSF allocation (not from the scratch
/// arena) because it survives across frames and draw calls.
pub struct PipelineState {
    /// Model-view-projection matrix [16] f32, column-major.
    pub mvp: GpuPtr,
    /// Light direction [3] f32 (toward-light, normalized).
    pub light_dir: GpuPtr,
    /// Light color [3] f32 (RGB intensity).
    pub light_color: GpuPtr,
    /// Ambient light [3] f32 (RGB).
    pub ambient: GpuPtr,
    /// Whether lighting is enabled for the current frame.
    pub lighting_enabled: bool,
    /// Viewport dimensions.
    pub viewport_w: u32,
    pub viewport_h: u32,
}

impl PipelineState {
    /// Allocate pipeline state buffers from TLSF.
    ///
    /// These are tiny (16 + 3 + 3 + 3 = 25 floats = 100 bytes) but
    /// persistent — they live for the lifetime of the render context.
    pub fn new(vp_w: u32, vp_h: u32, runtime: &Arc<PtxRuntime>) -> Result<Self> {
        let mvp = runtime.alloc(16 * 4)?; // 16 floats
        let light_dir = runtime.alloc(3 * 4)?;
        let light_color = runtime.alloc(3 * 4)?;
        let ambient = runtime.alloc(3 * 4)?;

        // Identity MVP
        let identity: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        unsafe {
            mvp.copy_from_host(
                identity.as_ptr() as *const libc::c_void,
                64,
            )?;
        }

        // Default light: overhead white
        let default_dir: [f32; 3] = [0.0, 1.0, 0.0];
        let default_color: [f32; 3] = [1.0, 1.0, 1.0];
        let default_ambient: [f32; 3] = [0.15, 0.15, 0.15];

        unsafe {
            light_dir.copy_from_host(default_dir.as_ptr() as *const _, 12)?;
            light_color.copy_from_host(default_color.as_ptr() as *const _, 12)?;
            ambient.copy_from_host(default_ambient.as_ptr() as *const _, 12)?;
        }

        Ok(Self {
            mvp,
            light_dir,
            light_color,
            ambient,
            lighting_enabled: false,
            viewport_w: vp_w,
            viewport_h: vp_h,
        })
    }

    /// Upload a new MVP matrix to the GPU.
    pub fn set_mvp(&self, matrix: &[f32; 16]) -> Result<()> {
        unsafe {
            self.mvp.copy_from_host(matrix.as_ptr() as *const _, 64)?;
        }
        Ok(())
    }

    /// Upload light parameters to the GPU.
    pub fn set_light(
        &mut self,
        dir: [f32; 3],
        color: [f32; 3],
        amb: [f32; 3],
    ) -> Result<()> {
        unsafe {
            self.light_dir.copy_from_host(dir.as_ptr() as *const _, 12)?;
            self.light_color.copy_from_host(color.as_ptr() as *const _, 12)?;
            self.ambient.copy_from_host(amb.as_ptr() as *const _, 12)?;
        }
        self.lighting_enabled = true;
        Ok(())
    }

    /// GPU pointer to the MVP matrix.
    #[inline]
    pub fn mvp_ptr(&self) -> *const f32 {
        self.mvp.as_ptr() as *const f32
    }

    /// GPU pointer to the light direction.
    #[inline]
    pub fn light_dir_ptr(&self) -> *const f32 {
        self.light_dir.as_ptr() as *const f32
    }

    /// GPU pointer to the light color.
    #[inline]
    pub fn light_color_ptr(&self) -> *const f32 {
        self.light_color.as_ptr() as *const f32
    }

    /// GPU pointer to the ambient color.
    #[inline]
    pub fn ambient_ptr(&self) -> *const f32 {
        self.ambient.as_ptr() as *const f32
    }
}
