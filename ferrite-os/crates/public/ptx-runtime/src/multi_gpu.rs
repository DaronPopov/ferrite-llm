//! Multi-GPU cluster management.
//!
//! Safe Rust wrappers around the PTX-OS multi-GPU C API, providing:
//! - GPU cluster initialization with automatic P2P setup
//! - Distributed tensors sharded across devices
//! - Cross-device data movement (migrate, broadcast, reduce)
//! - Load balancing across heterogeneous GPUs
//!
//! # Example
//!
//! ```no_run
//! use ptx_runtime::multi_gpu::GpuCluster;
//!
//! // Initialize a cluster with GPUs 0 and 1
//! let cluster = GpuCluster::new(&[0, 1]).expect("Failed to init cluster");
//! println!("Cluster has {} GPUs", cluster.num_devices());
//!
//! // Create a distributed tensor (1024x1024 f32, auto-sharded)
//! let tensor = cluster.create_tensor("weights", &[1024, 1024]).unwrap();
//!
//! // Broadcast from GPU 0 to all others
//! tensor.broadcast(0).unwrap();
//! ```

use std::ffi::CString;
use std::ptr;

use crate::error::{Error, Result};

// ============================================================================
// Reduction operation
// ============================================================================

/// Reduction operation for cross-device tensor reductions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
    Avg,
}

impl ReduceOp {
    fn as_cstr(&self) -> &'static [u8] {
        match self {
            ReduceOp::Sum => b"sum\0",
            ReduceOp::Max => b"max\0",
            ReduceOp::Min => b"min\0",
            ReduceOp::Avg => b"avg\0",
        }
    }
}

// ============================================================================
// Cluster statistics
// ============================================================================

/// Aggregate statistics for all GPUs in a cluster.
#[derive(Debug, Clone)]
pub struct ClusterStats {
    /// Per-device GPU utilization (0.0 - 1.0).
    pub utilization: Vec<f32>,
    /// Per-device memory usage fraction (0.0 - 1.0).
    pub memory_usage: Vec<f32>,
    /// P2P bandwidth matrix (GB/s). `p2p_bandwidth[i][j]` = bandwidth from device i to j.
    pub p2p_bandwidth: Vec<Vec<f32>>,
}

// ============================================================================
// GpuCluster
// ============================================================================

/// A multi-GPU cluster with automatic P2P setup and RAII cleanup.
///
/// Each device in the cluster gets its own `GPUHotRuntime` with TLSF allocator.
/// P2P peer access is automatically enabled between all capable device pairs.
pub struct GpuCluster {
    ptr: *mut ptx_sys::GPUMultiCluster,
    device_ids: Vec<i32>,
}

// The C API uses cudaSetDevice internally, so cluster operations must be
// externally synchronized. Individual device runtimes are thread-safe.
unsafe impl Send for GpuCluster {}
unsafe impl Sync for GpuCluster {}

impl GpuCluster {
    /// Initialize a multi-GPU cluster.
    ///
    /// Creates a `GPUHotRuntime` on each device, probes P2P capabilities,
    /// and enables peer access between all capable pairs.
    ///
    /// # Arguments
    ///
    /// * `device_ids` - GPU device IDs to include (e.g. `&[0, 1, 2, 3]`)
    pub fn new(device_ids: &[i32]) -> Result<Self> {
        if device_ids.is_empty() {
            return Err(Error::Internal {
                message: "GpuCluster requires at least one device".to_string(),
            });
        }
        if device_ids.len() > ptx_sys::GPU_HOT_MAX_GPUS {
            return Err(Error::Internal {
                message: format!(
                    "Too many devices: {} (max {})",
                    device_ids.len(),
                    ptx_sys::GPU_HOT_MAX_GPUS
                ),
            });
        }

        let ptr = unsafe {
            ptx_sys::gpu_multicluster_init(
                device_ids.as_ptr(),
                device_ids.len() as libc::c_int,
            )
        };

        if ptr.is_null() {
            return Err(Error::InitFailed {
                device_id: device_ids[0],
            });
        }

        Ok(Self {
            ptr,
            device_ids: device_ids.to_vec(),
        })
    }

    /// Get the number of devices in the cluster.
    pub fn num_devices(&self) -> usize {
        self.device_ids.len()
    }

    /// Get the device IDs in the cluster.
    pub fn device_ids(&self) -> &[i32] {
        &self.device_ids
    }

    /// Get the primary (first) device ID.
    pub fn primary_device(&self) -> i32 {
        self.device_ids[0]
    }

    /// Get aggregate cluster statistics.
    pub fn stats(&self) -> ClusterStats {
        let mut raw = ptx_sys::GPUMultiStats::default();
        unsafe {
            ptx_sys::gpu_multicluster_get_stats(self.ptr, &mut raw);
        }

        let n = self.device_ids.len();
        ClusterStats {
            utilization: raw.utilization[..n].to_vec(),
            memory_usage: raw.memory_usage[..n].to_vec(),
            p2p_bandwidth: (0..n)
                .map(|i| raw.p2p_bandwidth[i][..n].to_vec())
                .collect(),
        }
    }

    /// Measure P2P bandwidth between two devices (GB/s).
    pub fn measure_bandwidth(&self, src: i32, dst: i32) -> f32 {
        unsafe { ptx_sys::gpu_measure_bandwidth(src, dst) }
    }

    /// Distribute `total_work` units across devices proportional to their capacities.
    ///
    /// Returns a Vec with work assigned per device.
    pub fn balance_workload(&self, total_work: i32, capacities: &[i32]) -> Result<Vec<i32>> {
        if capacities.len() != self.device_ids.len() {
            return Err(Error::Internal {
                message: format!(
                    "capacities length {} != num_devices {}",
                    capacities.len(),
                    self.device_ids.len()
                ),
            });
        }

        let mut workload = vec![0i32; self.device_ids.len()];
        let rc = unsafe {
            ptx_sys::gpu_balance_workload(
                self.ptr,
                workload.as_mut_ptr(),
                total_work,
                capacities.as_ptr(),
            )
        };

        if rc != 0 {
            return Err(Error::Internal {
                message: "gpu_balance_workload failed".to_string(),
            });
        }

        Ok(workload)
    }

    /// Create a distributed tensor sharded across all cluster devices.
    ///
    /// The tensor is automatically partitioned across devices with roughly
    /// equal shard sizes.
    ///
    /// # Arguments
    ///
    /// * `name`  - Human-readable name for debugging
    /// * `shape` - Tensor dimensions (e.g. `&[1024, 1024]`)
    pub fn create_tensor(&self, name: &str, shape: &[i32]) -> Result<DistributedTensor> {
        self.create_tensor_with_dtype(name, shape, ptx_sys::MultiTensorDtype::Float32, None)
    }

    /// Create a distributed tensor with explicit dtype and optional device distribution.
    pub fn create_tensor_with_dtype(
        &self,
        name: &str,
        shape: &[i32],
        dtype: ptx_sys::MultiTensorDtype,
        device_distribution: Option<&[i32]>,
    ) -> Result<DistributedTensor> {
        let c_name = CString::new(name).map_err(|_| Error::Internal {
            message: "tensor name contains null byte".to_string(),
        })?;

        let dist_ptr = device_distribution
            .map(|d| d.as_ptr())
            .unwrap_or(ptr::null());

        let ptr = unsafe {
            ptx_sys::gpu_distributed_tensor_create(
                self.ptr,
                c_name.as_ptr(),
                shape.as_ptr(),
                shape.len() as libc::c_int,
                dtype,
                dist_ptr,
            )
        };

        if ptr.is_null() {
            return Err(Error::AllocationFailed {
                size: shape.iter().map(|&d| d as usize).product::<usize>() * 4,
            });
        }

        Ok(DistributedTensor {
            ptr,
            name: name.to_string(),
        })
    }

    /// Get the raw cluster pointer (for advanced FFI interop).
    ///
    /// # Safety
    ///
    /// The caller must not free or outlive the cluster.
    pub unsafe fn raw_ptr(&self) -> *mut ptx_sys::GPUMultiCluster {
        self.ptr
    }
}

impl Drop for GpuCluster {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ptx_sys::gpu_multicluster_shutdown(self.ptr);
            }
        }
    }
}

// ============================================================================
// DistributedTensor
// ============================================================================

/// A tensor distributed (sharded) across multiple GPUs.
///
/// Each device holds a contiguous shard of the data. Cross-device operations
/// (broadcast, reduce, migrate) use P2P copies when available.
pub struct DistributedTensor {
    ptr: *mut ptx_sys::DistributedTensor,
    name: String,
}

unsafe impl Send for DistributedTensor {}
unsafe impl Sync for DistributedTensor {}

impl DistributedTensor {
    /// Get the tensor name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Broadcast this tensor's data from `src_device` to all other devices.
    ///
    /// After broadcast, every shard contains a copy of the source shard's data.
    pub fn broadcast(&self, src_device: i32) -> Result<()> {
        let rc = unsafe { ptx_sys::gpu_broadcast_tensor(self.ptr, src_device) };
        if rc != 0 {
            return Err(Error::Internal {
                message: format!(
                    "broadcast of '{}' from device {} failed (rc={})",
                    self.name, src_device, rc
                ),
            });
        }
        Ok(())
    }

    /// Migrate a region of tensor data between two devices.
    ///
    /// # Arguments
    ///
    /// * `src` / `dst` - Source and destination device IDs
    /// * `offset` - Byte offset into the shard
    /// * `size` - Number of bytes to transfer
    pub fn migrate(&self, src: i32, dst: i32, offset: usize, size: usize) -> Result<()> {
        let rc = unsafe { ptx_sys::gpu_tensor_migrate(self.ptr, src, dst, offset, size) };
        if rc != 0 {
            return Err(Error::Internal {
                message: format!(
                    "migrate of '{}' from device {} to {} failed (rc={})",
                    self.name, src, dst, rc
                ),
            });
        }
        Ok(())
    }

    /// Reduce all shards into a single result tensor on `root_device`.
    ///
    /// The `result` tensor must already be allocated on the root device.
    ///
    /// # Arguments
    ///
    /// * `result` - Output tensor (must have a shard on `root_device`)
    /// * `root_device` - Device that receives the reduced result
    /// * `op` - Reduction operation (Sum, Max, Min, Avg)
    pub fn reduce_into(
        &self,
        result: &DistributedTensor,
        root_device: i32,
        op: ReduceOp,
    ) -> Result<()> {
        let op_str = op.as_cstr().as_ptr() as *const libc::c_char;
        let rc = unsafe {
            ptx_sys::gpu_reduce_tensor(result.ptr, self.ptr, root_device, op_str)
        };
        if rc != 0 {
            return Err(Error::Internal {
                message: format!(
                    "reduce of '{}' with {:?} to device {} failed (rc={})",
                    self.name, op, root_device, rc
                ),
            });
        }
        Ok(())
    }

    /// Get the raw tensor pointer (for advanced FFI interop).
    ///
    /// # Safety
    ///
    /// The caller must not free or outlive the tensor.
    pub unsafe fn raw_ptr(&self) -> *mut ptx_sys::DistributedTensor {
        self.ptr
    }
}

impl Drop for DistributedTensor {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ptx_sys::gpu_distributed_tensor_free(self.ptr);
            }
        }
    }
}

// ============================================================================
// P2P helpers (standalone, no cluster needed)
// ============================================================================

/// Enable peer-to-peer access between two GPUs (bidirectional).
///
/// This is called automatically by `GpuCluster::new`, but can be used
/// standalone for manual P2P setup.
pub fn enable_p2p(src: i32, dst: i32) -> Result<()> {
    let rc = unsafe { ptx_sys::gpu_enable_p2p(src, dst) };
    if rc != 0 {
        return Err(Error::Internal {
            message: format!("Failed to enable P2P between device {} and {}", src, dst),
        });
    }
    Ok(())
}

/// Disable peer-to-peer access between two GPUs.
pub fn disable_p2p(src: i32, dst: i32) -> Result<()> {
    let rc = unsafe { ptx_sys::gpu_disable_p2p(src, dst) };
    if rc != 0 {
        return Err(Error::Internal {
            message: format!("Failed to disable P2P between device {} and {}", src, dst),
        });
    }
    Ok(())
}

/// Measure P2P bandwidth between two devices (GB/s).
pub fn measure_bandwidth(src: i32, dst: i32) -> f32 {
    unsafe { ptx_sys::gpu_measure_bandwidth(src, dst) }
}
