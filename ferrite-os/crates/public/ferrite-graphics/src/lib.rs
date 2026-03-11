//! GPU software rasterization for Ferrite-OS.
//!
//! Provides 3D graphics as pure GPU compute — no OpenGL, Vulkan, or GPU
//! driver graphics stack required. Meshes, cameras, lights, and framebuffers
//! are all backed by [`ptx_tensor::Tensor`] and rendered via CUDA kernels
//! through the standard PTX-OS dispatch pipeline.
//!
//! # Pipeline
//!
//! ```text
//! Mesh vertices → MVP transform → perspective divide → viewport →
//! triangle rasterization (edge function + atomic depth) →
//! per-pixel Phong shading → packed RGBA → ferrite-window::present()
//! ```
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use ptx_runtime::PtxRuntime;
//! use ferrite_graphics::{Framebuffer, Mesh, Camera, Light};
//! use ferrite_window::{Window, WindowConfig};
//!
//! let runtime = Arc::new(PtxRuntime::new(0).unwrap());
//! let mut fb = Framebuffer::new(800, 600, &runtime).unwrap();
//! let mesh = Mesh::cube(1.0, &runtime).unwrap();
//! let camera = Camera::perspective(60.0_f32.to_radians(), 800.0 / 600.0, 0.1, 100.0)
//!     .look_at([0.0, 2.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
//! let light = Light::directional([0.5, -1.0, -0.5], [1.0, 1.0, 1.0]);
//!
//! let mut window = Window::new(WindowConfig::default()).unwrap();
//!
//! while window.is_open() {
//!     for event in window.poll_events() { /* handle */ }
//!     fb.clear([0.1, 0.1, 0.1, 1.0]);
//!     fb.draw_mesh(&mesh, &camera, Some(&light), &runtime).unwrap();
//!     let pixels = fb.to_rgba(&runtime).unwrap();
//!     window.present(&pixels, 800, 600).unwrap();
//! }
//! ```

mod camera;
mod framebuffer;
mod light;
mod mesh;
mod pipeline;

pub use camera::Camera;
pub use framebuffer::Framebuffer;
pub use light::Light;
pub use mesh::Mesh;

use std::fmt;

/// Graphics error.
#[derive(Debug)]
pub enum GraphicsError {
    /// GPU tensor/runtime error.
    Gpu(ptx_runtime::Error),
    /// Invalid mesh or geometry.
    InvalidGeometry(String),
    /// Framebuffer size mismatch.
    SizeMismatch(String),
}

impl fmt::Display for GraphicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gpu(e) => write!(f, "GPU error: {}", e),
            Self::InvalidGeometry(msg) => write!(f, "invalid geometry: {}", msg),
            Self::SizeMismatch(msg) => write!(f, "size mismatch: {}", msg),
        }
    }
}

impl std::error::Error for GraphicsError {}

impl From<ptx_runtime::Error> for GraphicsError {
    fn from(e: ptx_runtime::Error) -> Self {
        Self::Gpu(e)
    }
}

pub type Result<T> = std::result::Result<T, GraphicsError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_default_perspective() {
        let cam = Camera::perspective(
            std::f32::consts::FRAC_PI_4,
            16.0 / 9.0,
            0.1,
            100.0,
        );
        let mvp = cam.view_projection_matrix();
        assert_eq!(mvp.len(), 16);
        // Projection matrix should not be all zeros
        assert!(mvp.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn camera_look_at() {
        let cam = Camera::perspective(1.0, 1.0, 0.1, 100.0)
            .look_at([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let mvp = cam.view_projection_matrix();
        assert_eq!(mvp.len(), 16);
    }

    #[test]
    fn light_directions() {
        let l = Light::directional([0.0, -1.0, 0.0], [1.0, 1.0, 1.0]);
        let (dir, color, ambient) = l.params();
        assert_eq!(dir.len(), 3);
        assert_eq!(color.len(), 3);
        assert_eq!(ambient.len(), 3);
        // Light direction should be normalized
        let len: f32 = dir.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((len - 1.0).abs() < 0.01);
    }

    #[test]
    fn mesh_cube_geometry() {
        // Cube: 8 vertices, 12 triangles (2 per face * 6 faces)
        let positions = Mesh::cube_data();
        assert!(positions.0.len() > 0); // vertices
        assert!(positions.1.len() > 0); // indices
        assert_eq!(positions.1.len() % 3, 0); // triangles
    }
}
