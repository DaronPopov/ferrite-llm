#[cfg(feature = "cuda")]
pub use ug;
#[cfg(feature = "cuda")]
pub use ug_cuda;

#[cfg(feature = "cuda")]
pub fn create_device(device_index: usize) -> anyhow::Result<ug_cuda::runtime::Device> {
    Ok(ug_cuda::runtime::Device::new(device_index)?)
}
