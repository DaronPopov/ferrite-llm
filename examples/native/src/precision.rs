//! Precision utilities for model inference
//!
//! Provides dtype configuration for FP32, FP16, and BF16 inference.

use candle_core::{DType, Device};

/// Inference precision mode
#[derive(Clone, Copy, Debug)]
pub enum Precision {
    /// FP32 - Full precision, more accurate, slower
    F32,
    /// FP16 - Half precision, fast GPU inference (recommended for CUDA)
    F16,
    /// BF16 - Brain float16, fast GPU inference with better numeric stability
    BF16,
}

impl Precision {
    /// Convert precision to Candle DType
    pub fn dtype(&self) -> DType {
        match self {
            Precision::F32 => DType::F32,
            Precision::F16 => DType::F16,
            Precision::BF16 => DType::BF16,
        }
    }

    /// Select optimal precision based on device
    ///
    /// Returns FP16 for CUDA devices, FP32 for CPU
    pub fn from_device(device: &Device) -> Self {
        match device {
            Device::Cuda(_) => {
                println!("[Init] Using CUDA with FP16");
                Precision::F16
            }
            Device::Cpu => {
                println!("[Init] Using CPU with FP32");
                Precision::F32
            }
            _ => Precision::F32,
        }
    }

    /// Check if this precision requires conversion to F32 for sampling
    pub fn needs_f32_conversion(&self) -> bool {
        !matches!(self, Precision::F32)
    }
}
