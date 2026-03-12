use anyhow::{anyhow, Result};

#[derive(Debug, Clone, Copy)]
pub struct TorchTlsfInit {
    pub device_id: i32,
    pub pool_fraction: f32,
}

impl TorchTlsfInit {
    pub fn validate(self) -> Result<Self> {
        if self.device_id < 0 {
            return Err(anyhow!("device_id must be >= 0"));
        }
        if !(0.1..=0.9).contains(&self.pool_fraction) {
            return Err(anyhow!("pool_fraction must be in [0.1, 0.9]"));
        }
        Ok(self)
    }
}
