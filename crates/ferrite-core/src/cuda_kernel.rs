// CUDA Kernel Runtime - Framework for custom kernels
//
// This module provides the foundation for custom CUDA kernel integration
// Full runtime loading is a work in progress - see KERNEL_TEMPLATE.md for pattern

/// CUDA Kernel configuration
#[cfg(feature = "cuda")]
pub struct CudaKernelConfig {
    pub kernel_name: String,
    pub ptx_source: &'static str,
}

#[cfg(feature = "cuda")]
impl CudaKernelConfig {
    pub fn new(ptx_source: &'static str, kernel_name: &str) -> Self {
        Self {
            ptx_source,
            kernel_name: kernel_name.to_string(),
        }
    }
}

/// Kernel launch helper - Framework established
///
/// For actual kernel launching, see:
/// - flash_attention.rs for reference pattern
/// - KERNEL_TEMPLATE.md for step-by-step guide
#[cfg(feature = "cuda")]
pub struct CudaKernel {
    config: CudaKernelConfig,
}

#[cfg(feature = "cuda")]
impl CudaKernel {
    pub fn new(config: CudaKernelConfig) -> Self {
        Self { config }
    }

    pub fn kernel_name(&self) -> &str {
        &self.config.kernel_name
    }

    pub fn ptx_source(&self) -> &'static str {
        self.config.ptx_source
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_kernel_config() {
        const TEST_PTX: &str = "// Test PTX";
        let config = CudaKernelConfig::new(TEST_PTX, "test_kernel");
        assert_eq!(config.kernel_name, "test_kernel");
    }
}
