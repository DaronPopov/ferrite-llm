use std::sync::{Arc, Mutex};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};

const FL_INDEX_COUNT: usize = 32;
const SL_INDEX_COUNT_LOG2: usize = 5; 
const SL_INDEX_COUNT: usize = 1 << SL_INDEX_COUNT_LOG2;
const MIN_BLOCK_SIZE: usize = 64; 

#[link(name = "cuda")]
extern "C" {
    fn cuMemcpyHtoD_v2(dst: u64, src: *const std::ffi::c_void, bytes: usize) -> i32;
    fn cuMemcpyDtoH_v2(dst: *mut std::ffi::c_void, src: u64, bytes: usize) -> i32;
}

#[derive(Debug)]
pub struct TlsfAllocator {
    device: Arc<CudaDevice>,
    pool: CudaSlice<u8>,
    state: Mutex<AllocatorState>,
}

#[derive(Debug)]
struct AllocatorState {
    fl_bitmap: u32,
    sl_bitmaps: [u32; FL_INDEX_COUNT],
    current_pool_consumed: usize,
    total_capacity: usize,
}

impl TlsfAllocator {
    pub fn new(device: Arc<CudaDevice>, size_bytes: usize) -> Self {
        let pool = device.alloc_zeros::<u8>(size_bytes).expect("Failed to allocate TLSF hot pool");
        Self {
            device,
            pool,
            state: Mutex::new(AllocatorState {
                fl_bitmap: 0,
                sl_bitmaps: [0; FL_INDEX_COUNT],
                current_pool_consumed: 0,
                total_capacity: size_bytes,
            }),
        }
    }

    pub fn alloc(&self, mut size: usize) -> Option<usize> {
        size = size.max(MIN_BLOCK_SIZE);
        let mut state = self.state.lock().unwrap();

        let fl = 31 - (size as u32).leading_zeros() as usize;
        let sl = (size >> (fl - SL_INDEX_COUNT_LOG2)) & (SL_INDEX_COUNT - 1);

        if state.current_pool_consumed + size <= state.total_capacity {
            let offset = state.current_pool_consumed;
            state.current_pool_consumed += size;
            state.fl_bitmap |= 1 << fl;
            state.sl_bitmaps[fl] |= 1 << sl;
            Some(offset)
        } else {
            None
        }
    }

    /// Safe data transfer to pool offset
    pub fn copy_to_offset(&self, offset: usize, data: &[f32]) {
        let size_bytes = data.len() * 4;
        let dst_ptr = *self.pool.device_ptr() + offset as u64;
        let src_ptr = data.as_ptr() as *const std::ffi::c_void;
        
        unsafe {
            self.device.bind_to_thread().expect("Failed to bind CUDA context to thread");
            let res = cuMemcpyHtoD_v2(dst_ptr, src_ptr, size_bytes);
            if res != 0 {
                panic!("cuMemcpyHtoD_v2 failed: {}", res);
            }
        }
    }

    /// Safe data transfer from pool offset
    pub fn copy_from_offset(&self, offset: usize, out: &mut [f32]) {
        let size_bytes = out.len() * 4;
        let src_ptr = *self.pool.device_ptr() + offset as u64;
        let dst_ptr = out.as_mut_ptr() as *mut std::ffi::c_void;

        unsafe {
            self.device.bind_to_thread().expect("Failed to bind CUDA context to thread");
            let res = cuMemcpyDtoH_v2(dst_ptr, src_ptr, size_bytes);
            if res != 0 {
                panic!("cuMemcpyDtoH_v2 failed: {}", res);
            }
        }
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    pub fn pool_ptr(&self) -> u64 {
        *self.pool.device_ptr()
    }

    /// Returns the amount of memory currently consumed from the pool
    pub fn consumed(&self) -> usize {
        self.state.lock().unwrap().current_pool_consumed
    }

    /// Returns the total capacity of the pool
    pub fn capacity(&self) -> usize {
        self.state.lock().unwrap().total_capacity
    }

    /// Returns the amount of free memory available
    pub fn available(&self) -> usize {
        let state = self.state.lock().unwrap();
        state.total_capacity - state.current_pool_consumed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_allocator() -> TlsfAllocator {
        let device = CudaDevice::new(0).expect("No GPU found for tests");
        TlsfAllocator::new(device, 1024 * 1024) // 1MB pool
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_allocator_creation() {
        let alloc = create_test_allocator();
        assert_eq!(alloc.capacity(), 1024 * 1024);
        assert_eq!(alloc.consumed(), 0);
        assert_eq!(alloc.available(), 1024 * 1024);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_basic_allocation() {
        let alloc = create_test_allocator();

        let offset1 = alloc.alloc(256).expect("First allocation failed");
        assert_eq!(offset1, 0);
        assert!(alloc.consumed() >= 256);

        let offset2 = alloc.alloc(512).expect("Second allocation failed");
        assert!(offset2 >= 256);
        assert!(alloc.consumed() >= 768);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_min_block_size() {
        let alloc = create_test_allocator();

        // Allocating less than MIN_BLOCK_SIZE should still consume MIN_BLOCK_SIZE
        let _offset = alloc.alloc(1).expect("Allocation failed");
        assert!(alloc.consumed() >= MIN_BLOCK_SIZE);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_allocation_failure() {
        let device = CudaDevice::new(0).expect("No GPU found");
        let alloc = TlsfAllocator::new(device, 1024); // Small 1KB pool

        // This should fail - requesting more than capacity
        let result = alloc.alloc(2048);
        assert!(result.is_none());
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_copy_roundtrip() {
        let alloc = Arc::new(create_test_allocator());

        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let size_bytes = data.len() * 4;
        let offset = alloc.alloc(size_bytes).expect("Allocation failed");

        // Copy to device
        alloc.copy_to_offset(offset, &data);

        // Copy back from device
        let mut result = vec![0.0f32; 5];
        alloc.copy_from_offset(offset, &mut result);

        assert_eq!(data, result);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_multiple_copy_roundtrips() {
        let alloc = Arc::new(create_test_allocator());

        // First tensor
        let data1 = vec![1.0f32, 2.0, 3.0];
        let offset1 = alloc.alloc(data1.len() * 4).unwrap();
        alloc.copy_to_offset(offset1, &data1);

        // Second tensor
        let data2 = vec![10.0f32, 20.0, 30.0, 40.0];
        let offset2 = alloc.alloc(data2.len() * 4).unwrap();
        alloc.copy_to_offset(offset2, &data2);

        // Verify both
        let mut result1 = vec![0.0f32; 3];
        let mut result2 = vec![0.0f32; 4];
        alloc.copy_from_offset(offset1, &mut result1);
        alloc.copy_from_offset(offset2, &mut result2);

        assert_eq!(data1, result1);
        assert_eq!(data2, result2);
    }
}
