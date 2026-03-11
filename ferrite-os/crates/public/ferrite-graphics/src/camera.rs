//! Camera — view and projection matrices for 3D rendering.
//!
//! All matrices are stored column-major (OpenGL/CUDA convention).

/// A virtual camera that produces the combined view-projection matrix
/// needed by the vertex transform kernel.
#[derive(Debug, Clone)]
pub struct Camera {
    /// Combined view * projection in column-major order.
    vp: [f32; 16],
}

impl Camera {
    /// Create a perspective projection camera.
    ///
    /// - `fov_y` — vertical field of view in **radians**
    /// - `aspect` — width / height
    /// - `near`, `far` — clipping planes
    pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov_y * 0.5).tan();
        let nf = 1.0 / (near - far);

        // Column-major perspective matrix (maps to [-1,1] NDC)
        #[rustfmt::skip]
        let proj = [
            f / aspect, 0.0,  0.0,                    0.0,
            0.0,        f,    0.0,                    0.0,
            0.0,        0.0,  (far + near) * nf,     -1.0,
            0.0,        0.0,  2.0 * far * near * nf,  0.0,
        ];

        Self { vp: proj }
    }

    /// Create an orthographic projection camera.
    ///
    /// - `left`, `right`, `bottom`, `top` — frustum extents
    /// - `near`, `far` — clipping planes
    pub fn orthographic(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let rl = right - left;
        let tb = top - bottom;
        let fn_ = far - near;

        #[rustfmt::skip]
        let proj = [
            2.0 / rl,            0.0,                  0.0,                 0.0,
            0.0,                  2.0 / tb,             0.0,                 0.0,
            0.0,                  0.0,                  -2.0 / fn_,          0.0,
            -(right + left) / rl, -(top + bottom) / tb, -(far + near) / fn_, 1.0,
        ];

        Self { vp: proj }
    }

    /// Set the view transform via a look-at specification.
    ///
    /// Returns a new `Camera` with view * projection combined.
    pub fn look_at(mut self, eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> Self {
        let view = look_at_matrix(eye, center, up);
        self.vp = mat4_mul(&self.vp, &view);
        self
    }

    /// Get the combined view-projection matrix (16 floats, column-major).
    pub fn view_projection_matrix(&self) -> &[f32; 16] {
        &self.vp
    }
}

// ---------------------------------------------------------------------------
// Matrix math (column-major 4x4)
// ---------------------------------------------------------------------------

fn look_at_matrix(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> [f32; 16] {
    let f = normalize(sub(center, eye));
    let s = normalize(cross(f, up));
    let u = cross(s, f);

    #[rustfmt::skip]
    let view = [
         s[0],              u[0],             -f[0],             0.0,
         s[1],              u[1],             -f[1],             0.0,
         s[2],              u[2],             -f[2],             0.0,
        -dot(s, eye),      -dot(u, eye),       dot(f, eye),      1.0,
    ];
    view
}

/// Multiply two column-major 4x4 matrices: result = a * b.
fn mat4_mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0f32;
            for k in 0..4 {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            out[col * 4 + row] = sum;
        }
    }
    out
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        return [0.0, 0.0, 0.0];
    }
    let inv = 1.0 / len;
    [v[0] * inv, v[1] * inv, v[2] * inv]
}
