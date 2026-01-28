pub mod datasets;
pub mod checkpoint;

use std::sync::Arc;
use crate::dynamics::allocator::TlsfAllocator;

pub use datasets::{MnistDataset, MnistError, DataIterator};
pub use checkpoint::{Checkpoint, CheckpointMetadata, CheckpointError, save_grid, load_grid};

#[derive(Debug, Clone)]
pub enum Values {
    Host(Vec<f32>),
    Device {
        allocator: Arc<TlsfAllocator>,
        offset: usize,
        len: usize,
    },
}

impl Values {
    pub fn as_host(&self) -> Option<&Vec<f32>> {
        if let Values::Host(v) = self {
            Some(v)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct Grid {
    pub values: Values,
    pub shape: Vec<usize>,
}

impl Grid {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let total: usize = shape.iter().product();
        assert_eq!(data.len(), total, "Shape does not match data length");
        Grid {
            values: Values::Host(data),
            shape,
        }
    }

    pub fn to_device(&self, alloc: &Arc<TlsfAllocator>) -> Self {
        match &self.values {
            Values::Host(v) => {
                let size_bytes = v.len() * std::mem::size_of::<f32>();
                let offset = alloc.alloc(size_bytes).expect("TLSF Allocation failed");
                
                // Use safe allocator method
                alloc.copy_to_offset(offset, v);

                Grid {
                    values: Values::Device {
                        allocator: Arc::clone(alloc),
                        offset,
                        len: v.len(),
                    },
                    shape: self.shape.clone(),
                }
            }
            Values::Device { .. } => self.clone(),
        }
    }

    /// Returns the total number of elements in the grid
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns true if the grid is on the host (CPU)
    pub fn is_host(&self) -> bool {
        matches!(self.values, Values::Host(_))
    }

    /// Returns true if the grid is on the device (GPU)
    pub fn is_device(&self) -> bool {
        matches!(self.values, Values::Device { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let grid = Grid::new(data.clone(), vec![2, 3]);

        assert_eq!(grid.shape, vec![2, 3]);
        assert_eq!(grid.numel(), 6);
        assert!(grid.is_host());
        assert!(!grid.is_device());

        if let Values::Host(v) = &grid.values {
            assert_eq!(v, &data);
        } else {
            panic!("Expected Host values");
        }
    }

    #[test]
    fn test_grid_1d() {
        let data = vec![1.0, 2.0, 3.0];
        let grid = Grid::new(data, vec![3]);
        assert_eq!(grid.shape, vec![3]);
        assert_eq!(grid.numel(), 3);
    }

    #[test]
    fn test_grid_3d() {
        let data = vec![0.0; 24];
        let grid = Grid::new(data, vec![2, 3, 4]);
        assert_eq!(grid.shape, vec![2, 3, 4]);
        assert_eq!(grid.numel(), 24);
    }

    #[test]
    #[should_panic(expected = "Shape does not match data length")]
    fn test_grid_shape_mismatch() {
        let data = vec![1.0, 2.0, 3.0];
        Grid::new(data, vec![2, 3]); // 6 != 3, should panic
    }

    #[test]
    fn test_values_as_host() {
        let values = Values::Host(vec![1.0, 2.0]);
        assert!(values.as_host().is_some());
        assert_eq!(values.as_host().unwrap(), &vec![1.0, 2.0]);
    }

    #[test]
    fn test_grid_clone() {
        let grid = Grid::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let cloned = grid.clone();

        assert_eq!(grid.shape, cloned.shape);
        if let (Values::Host(v1), Values::Host(v2)) = (&grid.values, &cloned.values) {
            assert_eq!(v1, v2);
        }
    }
}
