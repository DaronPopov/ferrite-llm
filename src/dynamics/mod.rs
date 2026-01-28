pub mod allocator;
mod muscle_memory;
use std::sync::Arc;
pub use muscle_memory::MuscleMemory;
use crate::compute::synth::Synthesizer;
use allocator::TlsfAllocator;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    // Floating point types
    F16,
    BF16,
    F32,
    F64,

    // Quantized integer types (custom implementation)
    Int8,   // Signed 8-bit quantization
    UInt8,  // Unsigned 8-bit quantization
    Int4,   // Signed 4-bit quantization (packed, 2 values per byte)
    UInt4,  // Unsigned 4-bit quantization (packed, 2 values per byte)
}

impl Precision {
    /// Returns the size in bytes for this precision type
    /// Note: For 4-bit types, this returns 1 byte (which holds 2 values)
    pub fn size_bytes(&self) -> usize {
        match self {
            Precision::Int4 | Precision::UInt4 => 1,  // Packed: 2 values per byte
            Precision::Int8 | Precision::UInt8 => 1,
            Precision::F16 | Precision::BF16 => 2,
            Precision::F32 => 4,
            Precision::F64 => 8,
        }
    }

    /// Returns the size in bits for this precision type
    pub fn size_bits(&self) -> usize {
        match self {
            Precision::Int4 | Precision::UInt4 => 4,
            Precision::Int8 | Precision::UInt8 => 8,
            Precision::F16 | Precision::BF16 => 16,
            Precision::F32 => 32,
            Precision::F64 => 64,
        }
    }

    /// Returns the torch dtype string for this precision
    pub fn torch_dtype(&self) -> &'static str {
        match self {
            Precision::F16 => "float16",
            Precision::BF16 => "bfloat16",
            Precision::F32 => "float32",
            Precision::F64 => "float64",
            Precision::Int8 => "int8",
            Precision::UInt8 => "uint8",
            Precision::Int4 => "int4",   // Custom, not native torch
            Precision::UInt4 => "uint4", // Custom, not native torch
        }
    }

    /// Returns true if this is a half-precision floating point type
    pub fn is_half(&self) -> bool {
        matches!(self, Precision::F16 | Precision::BF16)
    }

    /// Returns true if this is a quantized integer type
    pub fn is_quantized(&self) -> bool {
        matches!(self, Precision::Int8 | Precision::UInt8 | Precision::Int4 | Precision::UInt4)
    }

    /// Returns true if this is a 4-bit packed type
    pub fn is_packed_4bit(&self) -> bool {
        matches!(self, Precision::Int4 | Precision::UInt4)
    }

    /// Returns true if this is a signed type
    pub fn is_signed(&self) -> bool {
        matches!(self, Precision::Int8 | Precision::Int4 | Precision::F16 | Precision::BF16 | Precision::F32 | Precision::F64)
    }

    /// Returns the min/max representable values for quantized types
    pub fn quant_range(&self) -> (f32, f32) {
        match self {
            Precision::Int8 => (-128.0, 127.0),
            Precision::UInt8 => (0.0, 255.0),
            Precision::Int4 => (-8.0, 7.0),
            Precision::UInt4 => (0.0, 15.0),
            _ => (f32::MIN, f32::MAX), // Float types
        }
    }

    /// Calculate storage bytes needed for n elements
    pub fn storage_bytes(&self, num_elements: usize) -> usize {
        match self {
            Precision::Int4 | Precision::UInt4 => (num_elements + 1) / 2, // 2 values per byte
            _ => num_elements * self.size_bytes(),
        }
    }
}

/// Quantization parameters for a tensor
#[derive(Debug, Clone, Copy)]
pub struct QuantParams {
    /// Scale factor: real_value = (quant_value - zero_point) * scale
    pub scale: f32,
    /// Zero point offset
    pub zero_point: i32,
    /// Precision type
    pub precision: Precision,
}

impl QuantParams {
    /// Create quantization params for symmetric quantization (zero_point = 0)
    pub fn symmetric(scale: f32, precision: Precision) -> Self {
        QuantParams { scale, zero_point: 0, precision }
    }

    /// Create quantization params for asymmetric quantization
    pub fn asymmetric(scale: f32, zero_point: i32, precision: Precision) -> Self {
        QuantParams { scale, zero_point, precision }
    }

    /// Compute quantization params from tensor min/max values
    pub fn from_range(min_val: f32, max_val: f32, precision: Precision) -> Self {
        let (qmin, qmax) = precision.quant_range();

        if precision.is_signed() {
            // Symmetric quantization for signed types
            let abs_max = min_val.abs().max(max_val.abs());
            let scale = abs_max / qmax;
            QuantParams { scale, zero_point: 0, precision }
        } else {
            // Asymmetric quantization for unsigned types
            let scale = (max_val - min_val) / (qmax - qmin);
            let zero_point = (qmin - min_val / scale).round() as i32;
            QuantParams { scale, zero_point, precision }
        }
    }
}

#[derive(Clone)]
pub enum Device {
    Cpu,
    Gpu(Arc<TlsfAllocator>),
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "Cpu"),
            Device::Gpu(_) => write!(f, "Gpu(TlsfAllocator)"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeRules {
    pub precision: Precision,
    pub device: Device,
    pub track_gradients: bool,
    pub muscle_memory: Arc<MuscleMemory>,
    pub synthesizer: Option<Arc<Synthesizer>>,
}

impl Default for RuntimeRules {
    fn default() -> Self {
        Self {
            precision: Precision::F32,
            device: Device::Cpu,
            track_gradients: false,
            muscle_memory: Arc::new(MuscleMemory::new()),
            synthesizer: None,
        }
    }
}
