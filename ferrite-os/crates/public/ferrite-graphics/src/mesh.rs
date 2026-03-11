//! Mesh — vertex + index data on the GPU.
//!
//! A mesh stores positions, optional normals and vertex colors as GPU tensors,
//! plus a triangle index buffer. All geometry lives in GPU memory and is
//! directly consumed by the rasterization kernels.

use std::sync::Arc;

use ptx_runtime::PtxRuntime;
use ptx_tensor::{DType, Tensor};

use crate::Result;

/// A triangle mesh stored on the GPU.
///
/// Positions are `[N, 3]` f32 tensors, indices are `[T, 3]` u32 tensors.
pub struct Mesh {
    /// Vertex positions [num_vertices, 3] — f32.
    pub positions: Tensor,
    /// Vertex normals [num_vertices, 3] — f32, or None.
    pub normals: Option<Tensor>,
    /// Per-vertex colors [num_vertices, 3] — f32, or None (defaults to white).
    pub colors: Option<Tensor>,
    /// Triangle indices [num_triangles, 3] — u32.
    pub indices: Tensor,
}

impl Mesh {
    /// Create a mesh from host-side vertex and index data.
    ///
    /// - `positions` — flat `[x0, y0, z0, x1, y1, z1, ...]`
    /// - `indices` — flat `[i0, i1, i2, ...]` (3 per triangle)
    pub fn from_data(
        positions: &[f32],
        indices: &[u32],
        normals: Option<&[f32]>,
        colors: Option<&[f32]>,
        runtime: &Arc<PtxRuntime>,
    ) -> Result<Self> {
        let num_verts = positions.len() / 3;
        let num_tris = indices.len() / 3;

        if positions.len() % 3 != 0 {
            return Err(crate::GraphicsError::InvalidGeometry(
                "positions length must be divisible by 3".into(),
            ));
        }
        if indices.len() % 3 != 0 {
            return Err(crate::GraphicsError::InvalidGeometry(
                "indices length must be divisible by 3".into(),
            ));
        }

        let pos_tensor =
            Tensor::from_slice(positions, &[num_verts, 3], DType::F32, runtime)?;
        let idx_tensor =
            Tensor::from_slice(indices, &[num_tris, 3], DType::U32, runtime)?;

        let normals_tensor = match normals {
            Some(n) => {
                if n.len() != positions.len() {
                    return Err(crate::GraphicsError::InvalidGeometry(
                        "normals must have same length as positions".into(),
                    ));
                }
                Some(Tensor::from_slice(n, &[num_verts, 3], DType::F32, runtime)?)
            }
            None => None,
        };

        let colors_tensor = match colors {
            Some(c) => {
                if c.len() != positions.len() {
                    return Err(crate::GraphicsError::InvalidGeometry(
                        "colors must have same length as positions".into(),
                    ));
                }
                Some(Tensor::from_slice(c, &[num_verts, 3], DType::F32, runtime)?)
            }
            None => None,
        };

        Ok(Self {
            positions: pos_tensor,
            normals: normals_tensor,
            colors: colors_tensor,
            indices: idx_tensor,
        })
    }

    /// Number of vertices.
    pub fn num_vertices(&self) -> usize {
        self.positions.shape()[0]
    }

    /// Number of triangles.
    pub fn num_triangles(&self) -> usize {
        self.indices.shape()[0]
    }

    /// Create a unit cube centered at the origin.
    pub fn cube(size: f32, runtime: &Arc<PtxRuntime>) -> Result<Self> {
        let (positions, normals, indices) = Self::cube_data();
        let scaled: Vec<f32> = positions.iter().map(|v| v * size).collect();
        Self::from_data(&scaled, &indices, Some(&normals), None, runtime)
    }

    /// Raw cube data: (positions, normals, indices).
    ///
    /// 24 vertices (4 per face for flat normals), 12 triangles.
    pub fn cube_data() -> (Vec<f32>, Vec<f32>, Vec<u32>) {
        let s = 0.5f32;

        #[rustfmt::skip]
        let positions: Vec<f32> = vec![
            // Front face (z = +s)
            -s, -s,  s,   s, -s,  s,   s,  s,  s,  -s,  s,  s,
            // Back face (z = -s)
             s, -s, -s,  -s, -s, -s,  -s,  s, -s,   s,  s, -s,
            // Top face (y = +s)
            -s,  s,  s,   s,  s,  s,   s,  s, -s,  -s,  s, -s,
            // Bottom face (y = -s)
            -s, -s, -s,   s, -s, -s,   s, -s,  s,  -s, -s,  s,
            // Right face (x = +s)
             s, -s,  s,   s, -s, -s,   s,  s, -s,   s,  s,  s,
            // Left face (x = -s)
            -s, -s, -s,  -s, -s,  s,  -s,  s,  s,  -s,  s, -s,
        ];

        #[rustfmt::skip]
        let normals: Vec<f32> = vec![
            // Front
            0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0,
            // Back
            0.0, 0.0,-1.0,  0.0, 0.0,-1.0,  0.0, 0.0,-1.0,  0.0, 0.0,-1.0,
            // Top
            0.0, 1.0, 0.0,  0.0, 1.0, 0.0,  0.0, 1.0, 0.0,  0.0, 1.0, 0.0,
            // Bottom
            0.0,-1.0, 0.0,  0.0,-1.0, 0.0,  0.0,-1.0, 0.0,  0.0,-1.0, 0.0,
            // Right
            1.0, 0.0, 0.0,  1.0, 0.0, 0.0,  1.0, 0.0, 0.0,  1.0, 0.0, 0.0,
            // Left
           -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0,
        ];

        #[rustfmt::skip]
        let indices: Vec<u32> = vec![
            // Front
             0,  1,  2,   0,  2,  3,
            // Back
             4,  5,  6,   4,  6,  7,
            // Top
             8,  9, 10,   8, 10, 11,
            // Bottom
            12, 13, 14,  12, 14, 15,
            // Right
            16, 17, 18,  16, 18, 19,
            // Left
            20, 21, 22,  20, 22, 23,
        ];

        (positions, normals, indices)
    }

    /// Create a triangle (single flat triangle for testing).
    pub fn triangle(runtime: &Arc<PtxRuntime>) -> Result<Self> {
        #[rustfmt::skip]
        let positions: Vec<f32> = vec![
             0.0,  0.5,  0.0,
            -0.5, -0.5,  0.0,
             0.5, -0.5,  0.0,
        ];
        let normals: Vec<f32> = vec![
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
        ];
        let indices: Vec<u32> = vec![0, 1, 2];

        Self::from_data(&positions, &indices, Some(&normals), None, runtime)
    }

    /// Create a quad (two triangles forming a square in the XY plane).
    pub fn quad(width: f32, height: f32, runtime: &Arc<PtxRuntime>) -> Result<Self> {
        let hw = width * 0.5;
        let hh = height * 0.5;

        #[rustfmt::skip]
        let positions: Vec<f32> = vec![
            -hw, -hh, 0.0,
             hw, -hh, 0.0,
             hw,  hh, 0.0,
            -hw,  hh, 0.0,
        ];
        let normals: Vec<f32> = vec![
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
        ];
        let indices: Vec<u32> = vec![0, 1, 2, 0, 2, 3];

        Self::from_data(&positions, &indices, Some(&normals), None, runtime)
    }
}

impl std::fmt::Debug for Mesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Mesh(vertices={}, triangles={}, normals={}, colors={})",
            self.num_vertices(),
            self.num_triangles(),
            self.normals.is_some(),
            self.colors.is_some(),
        )
    }
}
