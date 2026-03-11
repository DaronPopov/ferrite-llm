//! GPU device management.


/// Represents a CUDA GPU device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Device {
    id: i32,
}

impl Device {
    /// Create a device handle for the given device ID.
    pub fn new(id: i32) -> Self {
        Self { id }
    }

    /// Get the device ID.
    pub fn id(&self) -> i32 {
        self.id
    }

    /// Get the default device (device 0).
    pub fn default() -> Self {
        Self::new(0)
    }

    /// Check if this is the CPU device (used as marker, not actual CPU).
    pub fn is_cpu(&self) -> bool {
        self.id < 0
    }

    /// Check if this is a GPU device.
    pub fn is_gpu(&self) -> bool {
        self.id >= 0
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::new(0)
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_cpu() {
            write!(f, "cpu")
        } else {
            write!(f, "cuda:{}", self.id)
        }
    }
}

/// Device ID type alias for compatibility.
pub type DeviceId = i32;
