//! Light sources for Phong shading.

/// A light source for the shading pass.
#[derive(Debug, Clone)]
pub struct Light {
    /// Normalized light direction (pointing *toward* the light).
    direction: [f32; 3],
    /// Light color/intensity (RGB, typically [0..1] but HDR is fine).
    color: [f32; 3],
    /// Ambient light (RGB).
    ambient: [f32; 3],
}

impl Light {
    /// Create a directional light.
    ///
    /// `direction` — direction the light is shining (will be normalized
    /// and negated internally so the shader receives the "toward-light" vector).
    ///
    /// `color` — RGB intensity.
    pub fn directional(direction: [f32; 3], color: [f32; 3]) -> Self {
        let neg = [-direction[0], -direction[1], -direction[2]];
        let len = (neg[0] * neg[0] + neg[1] * neg[1] + neg[2] * neg[2]).sqrt();
        let dir = if len > 1e-10 {
            [neg[0] / len, neg[1] / len, neg[2] / len]
        } else {
            [0.0, 1.0, 0.0]
        };

        Self {
            direction: dir,
            color,
            ambient: [0.15, 0.15, 0.15],
        }
    }

    /// Set the ambient light level.
    pub fn with_ambient(mut self, ambient: [f32; 3]) -> Self {
        self.ambient = ambient;
        self
    }

    /// Get (direction, color, ambient) for the shading kernel.
    pub fn params(&self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        (self.direction, self.color, self.ambient)
    }
}

impl Default for Light {
    fn default() -> Self {
        Self::directional([0.3, -1.0, -0.5], [1.0, 1.0, 1.0])
    }
}
